import asyncio
import aiohttp
import random
import os
from dotenv import load_dotenv

class LLMClient:
    def __init__(self, model, url, api_key=None):
        load_dotenv()
        self.model = model
        self.url = url
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]

    def send(self, prompt, concurrent=20, timeout=30):
        original_prompts = prompt
        if isinstance(prompt, str):
            prompt = [prompt]
        elif not hasattr(prompt, '__iter__'):
            raise TypeError()

        prepped = self.prep(prompt)
        responses = asyncio.run(self._async_send(prepped, concurrent, timeout))

        if isinstance(original_prompts, str):
            return {"prompt": original_prompts, "response": responses[0]}
        else:
            return [{"prompt": p, "response": r} for p, r in zip(prompt, responses)]

    async def _async_send(self, messages, concurrent, timeout):
        semaphore = asyncio.Semaphore(concurrent)

        async def send_one(msg):
            async with semaphore:
                for attempt in range(3):
                    try:
                        async with aiohttp.ClientSession() as session:
                            headers = {}
                            if self.api_key:
                                headers["Authorization"] = f"Bearer {self.api_key}"
                            async with session.post(
                                self.url,
                                json=msg,
                                headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout)
                            ) as response:
                                if response.status < 400:
                                    return await response.json()
                                elif response.status < 500:
                                    raise aiohttp.ClientResponseError(
                                        request_info=response.request_info,
                                        history=response.history,
                                        status=response.status
                                    )
                                else:
                                    if attempt == 2:
                                        raise aiohttp.ClientResponseError(
                                            request_info=response.request_info,
                                            history=response.history,
                                            status=response.status
                                        )
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt == 2 or (isinstance(e, aiohttp.ClientResponseError) and 400 <= e.status < 500):
                            raise

                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)

        tasks = [send_one(msg) for msg in messages]
        return await asyncio.gather(*tasks)

    def prep_prompts(self, prompts):
        return [[self.prep(p)] for p in prompts]

    def prep(self, prompt):
        messages = [{"role":"user", "content": prompt}]
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2048,
            "stream": False,
            'provider': {
                'data_collection': 'deny',
                'sort': 'price'
            },
            'quantizations': 'bf16'
        }
        return data

    def __call__(self, prompt, **kwargs):
        return self.send(prompt, **kwargs)
