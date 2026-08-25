import asyncio
import websockets
import json

async def run():
    async with websockets.connect("ws://localhost:8001/api/v1/ws/analyze") as ws:
        await ws.send(json.dumps({"contact_id": "test-ws-001"}))
        print(await ws.recv())

        with open("sample.wav", "rb") as f:
            audio = f.read()

        chunk_size = 8192
        for i in range(0, len(audio), chunk_size):
            await ws.send(audio[i:i+chunk_size])
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                print(msg)
            except asyncio.TimeoutError:
                pass

        await ws.send(b"")
        await asyncio.sleep(5)
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            print(msg)
        except asyncio.TimeoutError:
            print("No final message")

if __name__ == "__main__":
    asyncio.run(run())