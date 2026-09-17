import asyncio
import time
import httpx

URL = "http://localhost:8000/api/v1/predict"
API_KEY = "IrisML-ApiKey-2026-Secure-0304"

payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


async def send_request(client):
    try:
        response = await client.post(
            URL,
            headers={"X-API-Key": API_KEY},
            json=payload,
        )
        return response.status_code
    except Exception as e:
        return f"ERROR: {type(e).__name__}"


async def main():
    total_requests = 50

    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *[send_request(client) for _ in range(total_requests)]
        )

    elapsed = time.perf_counter() - start

    successful = results.count(200)
    failed = total_requests - successful

    print(f"Total requests: {total_requests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Requests/second: {total_requests / elapsed:.2f}")


if __name__ == "__main__":
    asyncio.run(main())