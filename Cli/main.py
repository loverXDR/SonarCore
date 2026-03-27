"""Command Line Interface for Sonar Core"""

import asyncio
import os
import sys

from Api.dependencies import get_agent_manager


async def async_main():
    """Main asynchronous loop for the CLI interface."""
    manager = get_agent_manager()
    
    while True:
        print("\n=== SonarCore CLI ===")
        print("1. Transcribe Audio (Raw ASR)")
        print("2. Initialize Chat Agent from Audio")
        print("3. Initialize Chat Agent from Text")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            await handle_transcribe(manager)
        elif choice == '2':
            await handle_init_audio(manager)
        elif choice == '3':
            await handle_init_text(manager)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


async def handle_transcribe(manager):
    """Handle raw transcription flow."""
    path = input("Enter path to audio file: ").strip()
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        return
        
    diarize_input = input("Enable diarization? (y/N): ").strip().lower()
    use_diarization = diarize_input in ['y', 'yes']
    
    print(f"\n[INFO] Transcribing {'with' if use_diarization else 'without'} diarization... This may take a while.")
    try:
        result = await manager.transcribe_audio(path, use_diarization=use_diarization)
        print("\n--- Transcription Result ---")
        print(result["text"])
        print("----------------------------\n")
    except Exception as e:
        print(f"An error occurred: {e}")


async def handle_init_audio(manager):
    """Handle initializing a session from audio."""
    path = input("Enter path to audio file: ").strip()
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        return
        
    diarize_input = input("Enable diarization? (y/N): ").strip().lower()
    use_diarization = diarize_input in ['y', 'yes']
    
    print(f"\n[INFO] Initializing session... This involves ASR and Indexing, please wait.")
    try:
        session_id = await manager.create_session_from_audio(path, use_diarization=use_diarization)
        print(f"\n[SUCCESS] Session '{session_id}' created successfully!")
        await chat_loop(manager, session_id)
    except Exception as e:
        print(f"An error occurred: {e}")


async def handle_init_text(manager):
    """Handle initializing a session from text."""
    print("Enter raw text (type '/done' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line.strip() == '/done':
            break
        lines.append(line)
        
    text = "\n".join(lines).strip()
    if not text:
        print("Error: Empty text provided.")
        return
        
    print(f"\n[INFO] Initializing session from text... Please wait.")
    try:
        session_id = await manager.create_session_from_text(text)
        print(f"\n[SUCCESS] Session '{session_id}' created successfully!")
        await chat_loop(manager, session_id)
    except Exception as e:
        print(f"An error occurred: {e}")


async def chat_loop(manager, session_id: str):
    """Interactive loop for chat with the agent."""
    print("\n=== Chat Session ===")
    print(f"Session ID: {session_id}")
    print("Type '/exit' or '/quit' to leave the chat loop and return to the main menu.")
    
    while True:
        try:
            msg = input("\nYou: ").strip()
            if not msg:
                continue
            if msg.lower() in ['/exit', '/quit']:
                print("Leaving chat session...\n")
                break
                
            print("Agent is typing...")
            response = await manager.chat(session_id, msg)
            print(f"Agent: {response.answer}")
        except KeyboardInterrupt:
            print("\nLeaving chat session...")
            break
        except Exception as e:
            print(f"Error occurred during chat: {e}")
            
    # Optionally delete the session when leaving the chat to save memory
    del_input = input("Delete this session from memory? (Y/n): ").strip().lower()
    if del_input not in ['n', 'no']:
        manager.delete_session(session_id)
        print("Session deleted.")


def main():
    """Application entry point."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nProgram terminated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
