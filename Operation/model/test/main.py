import asyncio
from pathlib import Path

from chat import ChatConfig, ChatSession


async def test_async(session: ChatSession):
    ans = await session.ask_async("Hello World!")
    print(ans)


keyset_pth = Path(__file__).parent / "keyset.yml"
service = ChatConfig(keyset_pth)[0]
print(service)
print(">>>>>>> SYNC")
session = ChatSession(service, "You are a helpful assistant.")
ans = session.ask("What is your name?" * 10)
print(ans)
print(">>>>>>> ASYNC")
asyncio.run(test_async(session))
