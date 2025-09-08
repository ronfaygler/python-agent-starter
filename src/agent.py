import logging
import os
import pathlib

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
    ChatContext,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero, groq
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from recording import RecordingManager

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# Helper function to load files from a directory
async def load_files_from_directory(directory_path):
    try:
        file_contents = []
        directory = pathlib.Path(directory_path)
        
        # Create directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)
        
        for file_path in directory.glob('*.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    file_contents.append(content)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        return file_contents
    except Exception as e:
        logger.error(f"Error loading files from {directory_path}: {e}")
        return []


# Function to load instructions from files
async def load_instructions():
    try:
        instructions_dir = pathlib.Path.cwd() / 'docs' / 'instructions'
        logger.info(f'Loading instructions from: {instructions_dir}')
        
        instruction_files = await load_files_from_directory(instructions_dir)
        if instruction_files:
            logger.info(f"Loaded {len(instruction_files)} instruction files")
            return '\n\n'.join(instruction_files)
        else:
            logger.info('No instruction files found, using default')
            return 'You are a helpful voice AI assistant. Respond in Hebrew.'
    except Exception as e:
        logger.error(f'Error loading instructions: {e}')
        return 'You are a helpful voice AI assistant. Respond in Hebrew.'


# Function to load knowledge files
async def load_knowledge():
    try:
        knowledge_dir = pathlib.Path.cwd() / 'docs' / 'knowledge'
        logger.info(f'Loading knowledge from: {knowledge_dir}')
        
        knowledge_files = await load_files_from_directory(knowledge_dir)
        if knowledge_files:
            logger.info(f"Loaded {len(knowledge_files)} knowledge files")
            return '\n\n---\n\n'.join(knowledge_files)
        else:
            logger.info('No knowledge files found')
            return ''
    except Exception as e:
        logger.error(f'Error loading knowledge files: {e}')
        return ''


# Initialize default values
default_instructions = None
default_knowledge = None


class Assistant(Agent):
    def __init__(self, chat_ctx: ChatContext, instructions: str) -> None:
        super().__init__(chat_ctx=chat_ctx, instructions=instructions)

    # # all functions annotated with @function_tool will be passed to the LLM when this
    # # agent is active
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.

    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """

    #     logger.info(f"Looking up weather for {location}")

    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info('========== AGENT ENTRY FUNCTION STARTED ==========')

    # language = 'hebrew'
    # Load instructions and knowledge before proceeding
    logger.info('Loading instructions and knowledge...')
    instructions = await load_instructions()
    knowledge_content = await load_knowledge()
    
    logger.info(f"Instructions loaded: {instructions[:50]}...")
    if instructions:
        logger.info(f"Instructions last part: {instructions[-50:]}...")
    
    logger.info(f"Knowledge loaded: {knowledge_content[:50]}...")
    if knowledge_content:
        logger.info(f"Knowledge last part: {knowledge_content[-50:]}...")
    
    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(model="gpt-4o-mini"),
        # llm=groq.LLM(
        #     model="llama-3.1-8b-instant"
        # ),
        # llm=openai.LLM.with_ollama(
        #     model="llama3.1",
        #     base_url="http://localhost:11434/v1",
        # ),
        # llm=openai.LLM.with_together(
        #     model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        # ),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        # stt=deepgram.STT(model="nova-3", language="multi"),
        # stt=groq.STT(model='whisper-large-v3-turbo', language="multi"),
        stt=openai.STT(model="whisper-1"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        # tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        tts=openai.TTS(model="gpt-4o-mini-tts"),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # if knowledge_content:
    #     # This will need to be handled differently depending on your specific needs
    #     # You might need to use a different approach to add knowledge to the context
    #     logger.info("Knowledge content available - will be used by the agent")
    

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession (
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Create a chat context with the knowledge content
    chat_ctx = ChatContext()
    if knowledge_content:
        chat_ctx.add_message(role="assistant", content=f"Reference information:\n{knowledge_content}")

    # Start the session
    await session.start(
        agent=Assistant(chat_ctx=chat_ctx, instructions=instructions),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

    # Initialize recording manager
    recording_manager = RecordingManager()
    recording_id = None

    try:
        # Start recording the room
        recording_id = await recording_manager.start_recording(ctx.room.name)
        if recording_id:
            logger.info(f"Started recording with ID: {recording_id}")
        else:
            logger.warning("Failed to start recording")
    except Exception as e:
        logger.error(f"Error starting recording: {e}")

    # Add shutdown callback to stop recording when session ends
    async def cleanup():
        if recording_id:
            logger.info("Stopping recording...")
            await recording_manager.stop_recording()
        await recording_manager.close()
    
    ctx.add_shutdown_callback(cleanup)

    # Generate initial greeting
    await session.generate_reply(
        instructions="תברך את המשתמש עם השם שלו בשפה העברית בלבד. נא תשתמש לאורך כל השיחה בשפה העברית.",
    )
    
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
