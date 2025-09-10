import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, cast
import json

from livekit import api
from livekit.api import StopEgressRequest

logger = logging.getLogger("recording")

class RecordingManager:
    def __init__(self):
        self._livekit_api: Optional[api.LiveKitAPI] = None
        self._current_recording_id: Optional[str] = None
        self._s3_client = None
        self._init_livekit_api()

    def _init_livekit_api(self) -> None:
        livekit_url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")

        if not all([livekit_url, api_key, api_secret]):
            logger.warning("Missing LiveKit credentials. Recording will be disabled.")
            return

        self._livekit_api = api.LiveKitAPI(
            livekit_url,
            api_key=api_key,
            api_secret=api_secret
        )

    async def _ensure_public_access(self, bucket_name: str, region: str) -> None:
        """Ensure the bucket has a public read policy"""
        import boto3
        from botocore.client import Config
        
        if not self._s3_client:
            self._s3_client = boto3.client(
                's3',
                endpoint_url=f'https://{region}.digitaloceanspaces.com',
                aws_access_key_id=os.getenv("DO_SPACES_KEY", ""),
                aws_secret_access_key=os.getenv("DO_SPACES_SECRET", ""),
                config=Config(signature_version='s3v4')
            )
        
        # Define the bucket policy
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "PublicReadGetObject",
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{bucket_name}/*"
                }
            ]
        }
        
        # Convert the policy to a JSON string
        policy_string = json.dumps(bucket_policy)
        
        # Set the new policy
        self._s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=policy_string
        )
        logger.info(f"Set public read policy for bucket: {bucket_name}")

    async def start_recording(self, room_name: str, modelsNames: list[str]) -> Optional[str]:
        if not self._livekit_api:
            logger.warning("Cannot start recording: LiveKit API not initialized")
            return None

        try:
            # Get Digital Ocean Spaces configuration
            endpoint = os.getenv("DO_SPACES_ENDPOINT", "")
            bucket_name = os.getenv("DO_SPACES_BUCKET", "")
            region = endpoint.split('.')[0]
            
            # Ensure the bucket has public read access
            await self._ensure_public_access(bucket_name, region)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            filename_prefix = f"{room_name}-{modelsNames[0]}_{modelsNames[1]}_{modelsNames[2]}-{timestamp}"
            bucket_name = os.getenv("DO_SPACES_BUCKET", "")
            region = endpoint.split('.')[0]
            # Main recording file path
            s3_path = f"https://{bucket_name}.{endpoint}/{filename_prefix}.mp4"
            # HLS playlist path
            # hls_playlist = f"https://{bucket_name}.{endpoint}/{filename_prefix}.m3u8"
            logger.info(f"Recording will be saved to:")
            logger.info(f"- Main file: {s3_path}")
            # logger.info(f"- HLS Playlist: {hls_playlist}")

            # Create recording request
            request = api.RoomCompositeEgressRequest(
                room_name=room_name,
                layout="speaker",
                preset=api.EncodingOptionsPreset.H264_720P_30,
                audio_only=False,
                file_outputs=[
                    api.EncodedFileOutput(
                        file_type=api.EncodedFileType.MP4,
                        filepath=f"{filename_prefix}.mp4",
                        s3=api.S3Upload(
                            access_key=os.getenv("DO_SPACES_KEY", ""),
                            secret=os.getenv("DO_SPACES_SECRET", ""),
                            region=region,
                            bucket=bucket_name,
                            endpoint=f"https://{endpoint}",
                            force_path_style=True
                        ),
                    )
                ]
            )

            logger.info(f"Starting recording for room: {room_name}")
            response = await self._livekit_api.egress.start_room_composite_egress(request)
            self._current_recording_id = response.egress_id
            logger.info(f"Recording started with ID: {self._current_recording_id}")
            return self._current_recording_id

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return None

    async def stop_recording(self) -> None:
        if not self._livekit_api or not self._current_recording_id:
            return

        try:
            # First check the current status of the recording
            # status = await self.get_recording_status()
            # if status and status.get('status') in ['EGRESS_ENDING', 'EGRESS_COMPLETE', 'EGRESS_FAILED', 'EGRESS_ABORTED']:
            #     logger.info(f"Recording {self._current_recording_id} is already in final state: {status.get('status')}")
            #     return

            logger.info(f"Stopping recording with ID: {self._current_recording_id}")
            
            try:
                # First try to stop the recording
                response = await self._livekit_api.egress.stop_egress(
                    api.StopEgressRequest(egress_id=self._current_recording_id)
                )
                
                # Log the file paths from the response if available
                if hasattr(response, 'to_dict'):
                    response_dict = response.to_dict()
                    if 'file_results' in response_dict:
                        for file_result in response_dict['file_results']:
                            if 'filename' in file_result:
                                logger.info(f"Recording saved to: {file_result['filename']}")
                            if 'playlist_name' in file_result:
                                logger.info(f"Playlist file: {file_result['playlist_name']}")
                    elif 'file' in response_dict and response_dict['file']:
                        logger.info(f"Recording file: {response_dict['file']}")
                    elif 'playlist' in response_dict and response_dict['playlist']:
                        logger.info(f"Playlist file: {response_dict['playlist']}")
                
                logger.info("Recording stopped successfully")
                
            except Exception as stop_error:
                # If stopping fails with failed_precondition, check if it's already in a final state
                if 'failed_precondition' in str(stop_error) and 'EGRESS_FAILED' in str(stop_error):
                    logger.info("Recording has already failed, checking for any saved files...")
                    try:
                        # List all egresses and find our recording
                        logger.debug("Calling list_egress...")
                        egress_list = await self._livekit_api.egress.list_egress(
                            list=api.ListEgressRequest()
                        )
                        logger.debug(f"list_egress response type: {type(egress_list)}")
                        logger.debug(f"list_egress response: {egress_list}")
                        
                        if egress_list:
                            # Try different ways to access the items
                            items = []
                            if hasattr(egress_list, 'items'):
                                items = egress_list.items
                            elif isinstance(egress_list, list):
                                items = egress_list
                                
                            logger.debug(f"Found {len(items)} egress items")
                            
                            for egress in items:
                                logger.debug(f"Processing egress item: {egress}")
                                egress_id = getattr(egress, 'egress_id', None) or getattr(egress, 'id', None)
                                if egress_id == self._current_recording_id:
                                    logger.debug(f"Found matching egress: {egress_id}")
                                    try:
                                        if hasattr(egress, 'to_dict'):
                                            egress_dict = egress.to_dict()
                                            logger.debug(f"Egress dict: {egress_dict}")
                                            
                                            # Check for file path in different possible locations
                                            file_path = None
                                            
                                            # Check for file in room_composite
                                            if 'room_composite' in egress_dict and 'file_outputs' in egress_dict['room_composite']:
                                                for output in egress_dict['room_composite']['file_outputs']:
                                                    if 'filepath' in output:
                                                        file_path = output['filepath']
                                                        break
                                            
                                            # Check for direct file info
                                            if not file_path and 'file' in egress_dict and 'filename' in egress_dict['file']:
                                                file_path = egress_dict['file']['filename']
                                            
                                            if file_path:
                                                logger.info(f"Recording file: {file_path}")
                                                
                                                # If there's an S3 error, log the local path that was attempted
                                                if 'error' in str(egress_dict) and 'S3 upload failed' in str(egress_dict):
                                                    logger.warning(f"S3 upload failed. File was saved locally at: {file_path}")
                                            
                                            # Log any error details
                                            if 'error' in str(egress_dict):
                                                logger.warning(f"Recording error: {egress_dict.get('error', 'Unknown error')}")
                                                
                                    except Exception as e:
                                        logger.debug(f"Error processing egress info: {e}")
                    except Exception as info_error:
                        logger.debug(f"Could not list recordings: {info_error}")
                else:
                    logger.error(f"Error while stopping recording: {stop_error}")
                    
        except Exception as e:
            logger.error(f"Unexpected error while stopping recording: {e}")
        finally:
            self._current_recording_id = None

    # async def get_recording_status(self) -> Optional[Dict[str, Any]]:
    #     if not self._livekit_api or not self._current_recording_id:
    #         return None

    #     try:
    #         # First try to get the specific egress info
    #         try:
    #             egress_info = await self._livekit_api.egress.get_egress_info(
    #                 egress_id=self._current_recording_id
    #             )
    #             if egress_info:
    #                 return egress_info.to_dict()
    #         except Exception as e:
    #             logger.debug(f"Could not get specific egress info: {e}")
    #             # Fall through to list_egress

    #         # Fallback to listing all egresses if get_egress_info fails
    #         egress_list = await self._livekit_api.egress.list_egress()
    #         if egress_list:
    #             # Find our specific egress in the list
    #             for egress in egress_list:
    #                 if egress.egress_id == self._current_recording_id:
    #                     return egress.to_dict()
    #         return None
    #     except Exception as e:
    #         logger.error(f"Failed to get recording status: {e}")
    #         return None

    async def close(self):
        # No explicit cleanup needed for LiveKitAPI
        pass