import asyncio
import os
from gemini_webapi import GeminiClient

SECURE_1PSID = os.environ.get("SECURE_1PSID", "")
SECURE_1PSIDTS = os.environ.get("SECURE_1PSIDTS", "")

async def main():
    if not SECURE_1PSID or not SECURE_1PSIDTS:
        print("Missing credentials.")
        return

    client = GeminiClient(SECURE_1PSID, SECURE_1PSIDTS)
    await client.init(timeout=30)
    
    print("Sending message...")
    response = await client.generate_content("Say 'hello world' and nothing else.")
    
    print("Response text:", response.text)
    print("Dir response:", dir(response))
    if hasattr(response, 'metadata'):
        print("Metadata:", response.metadata)
        if isinstance(response.metadata, list) and len(response.metadata) > 0:
            cid = response.metadata[0]
            print(f"CID extracted: {cid}")
            try:
                print(f"Deleting chat {cid}...")
                await client.delete_chat(cid)
                print("Deleted.")
            except Exception as e:
                print("Error deleting:", e)

    print("\nTesting stream...")
    async for chunk in client.generate_content_stream("Say 'hello world streaming' and nothing else."):
        if hasattr(chunk, 'metadata'):
            print("Stream chunk metadata:", chunk.metadata)
            if chunk.metadata and len(chunk.metadata) > 0:
                cid = chunk.metadata[0]
                print(f"CID from stream: {cid}")
                # We can delete it later. Let's just print it.
                break

if __name__ == "__main__":
    asyncio.run(main())
