import { afterEach, beforeEach, expect, test } from 'bun:test'
import { createOpenAIShimClient } from './openaiShim.ts'

type FetchType = typeof globalThis.fetch

const originalEnv = {
  OPENAI_BASE_URL: process.env.OPENAI_BASE_URL,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
}

const originalFetch = globalThis.fetch

type OpenAIShimClient = {
  beta: {
    messages: {
      create: (
        params: Record<string, unknown>,
        options?: Record<string, unknown>,
      ) => Promise<unknown> & {
        withResponse: () => Promise<{ data: AsyncIterable<Record<string, unknown>> }>
      }
    }
  }
}

function makeSseResponse(lines: string[]): Response {
  const encoder = new TextEncoder()
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const line of lines) {
          controller.enqueue(encoder.encode(line))
        }
        controller.close()
      },
    }),
    {
      headers: {
        'Content-Type': 'text/event-stream',
      },
    },
  )
}

function makeStreamChunks(chunks: unknown[]): string[] {
  return [
    ...chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`),
    'data: [DONE]\n\n',
  ]
}

async function collectEvents(stream: AsyncIterable<Record<string, unknown>>): Promise<Array<Record<string, unknown>>> {
  const events: Array<Record<string, unknown>> = []
  for await (const event of stream) {
    events.push(event)
  }
  return events
}

beforeEach(() => {
  process.env.OPENAI_BASE_URL = 'http://example.test/v1'
  process.env.OPENAI_API_KEY = 'test-key'
})

afterEach(() => {
  process.env.OPENAI_BASE_URL = originalEnv.OPENAI_BASE_URL
  process.env.OPENAI_API_KEY = originalEnv.OPENAI_API_KEY
  globalThis.fetch = originalFetch
})

test('preserves usage from final OpenAI stream chunk with empty choices', async () => {
  globalThis.fetch = (async (_input, init) => {
    const url = typeof _input === 'string' ? _input : _input.url
    expect(url).toBe('http://example.test/v1/chat/completions')

    const body = JSON.parse(String(init?.body))
    expect(body.stream).toBe(true)
    expect(body.stream_options).toEqual({ include_usage: true })

    const chunks = makeStreamChunks([
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: { role: 'assistant', content: 'hello world' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'stop',
          },
        ],
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [],
        usage: {
          prompt_tokens: 123,
          completion_tokens: 45,
          total_tokens: 168,
        },
      },
    ])

    return makeSseResponse(chunks)
  }) as FetchType

  const client = createOpenAIShimClient({}) as OpenAIShimClient

  const result = await client.beta.messages
    .create({
      model: 'fake-model',
      system: 'test system',
      messages: [{ role: 'user', content: 'hello' }],
      max_tokens: 64,
      stream: true,
    })
    .withResponse()

  const events: Array<Record<string, unknown>> = []
  for await (const event of result.data) {
    events.push(event)
  }

  const usageEvent = events.find(
    event => event.type === 'message_delta' && typeof event.usage === 'object' && event.usage !== null,
  ) as { usage?: { input_tokens?: number; output_tokens?: number } } | undefined

  expect(usageEvent).toBeDefined()
  expect(usageEvent?.usage?.input_tokens).toBe(123)
  expect(usageEvent?.usage?.output_tokens).toBe(45)
})

test('preserves Gemini tool call extra_content in follow-up requests', async () => {
  let requestBody: Record<string, unknown> | undefined

  globalThis.fetch = (async (_input, init) => {
    requestBody = JSON.parse(String(init?.body))

    return new Response(
      JSON.stringify({
        id: 'chatcmpl-1',
        model: 'google/gemini-3.1-pro-preview',
        choices: [
          {
            message: {
              role: 'assistant',
              content: 'done',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 12,
          completion_tokens: 4,
          total_tokens: 16,
        },
      }),
      {
        headers: {
          'Content-Type': 'application/json',
        },
      },
    )
  }) as FetchType

  const client = createOpenAIShimClient({}) as OpenAIShimClient

  await client.beta.messages.create({
    model: 'google/gemini-3.1-pro-preview',
    system: 'test system',
    messages: [
      { role: 'user', content: 'Use Bash' },
      {
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: 'call_1',
            name: 'Bash',
            input: { command: 'pwd' },
            extra_content: {
              google: {
                thought_signature: 'sig-123',
              },
            },
          },
        ],
      },
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_1',
            content: 'D:\\repo',
          },
        ],
      },
    ],
    max_tokens: 64,
    stream: false,
  })

  const assistantWithToolCall = (requestBody?.messages as Array<Record<string, unknown>>).find(
    message => Array.isArray(message.tool_calls),
  ) as { tool_calls?: Array<Record<string, unknown>> } | undefined

  expect(assistantWithToolCall?.tool_calls?.[0]).toMatchObject({
    id: 'call_1',
    type: 'function',
    function: {
      name: 'Bash',
      arguments: JSON.stringify({ command: 'pwd' }),
    },
    extra_content: {
      google: {
        thought_signature: 'sig-123',
      },
    },
  })
})

test('preserves Gemini tool call extra_content from streaming chunks', async () => {
  globalThis.fetch = (async (_input, _init) => {
    const chunks = makeStreamChunks([
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'google/gemini-3.1-pro-preview',
        choices: [
          {
            index: 0,
            delta: {
              role: 'assistant',
              tool_calls: [
                {
                  index: 0,
                  id: 'function-call-1',
                  type: 'function',
                  extra_content: {
                    google: {
                      thought_signature: 'sig-stream',
                    },
                  },
                  function: {
                    name: 'Bash',
                    arguments: '{"command":"pwd"}',
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'google/gemini-3.1-pro-preview',
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'tool_calls',
          },
        ],
      },
    ])

    return makeSseResponse(chunks)
  }) as FetchType

  const client = createOpenAIShimClient({}) as OpenAIShimClient

  const result = await client.beta.messages
    .create({
      model: 'google/gemini-3.1-pro-preview',
      system: 'test system',
      messages: [{ role: 'user', content: 'Use Bash' }],
      max_tokens: 64,
      stream: true,
    })
    .withResponse()

  const events = await collectEvents(result.data)

  const toolStart = events.find(
    event =>
      event.type === 'content_block_start' &&
      typeof event.content_block === 'object' &&
      event.content_block !== null &&
      (event.content_block as Record<string, unknown>).type === 'tool_use',
  ) as { content_block?: Record<string, unknown> } | undefined

  expect(toolStart?.content_block).toMatchObject({
    type: 'tool_use',
    id: 'function-call-1',
    name: 'Bash',
    extra_content: {
      google: {
        thought_signature: 'sig-stream',
      },
    },
  })
})

test('releases OpenAI stream reader lock after iteration stops early', async () => {
  let streamingResponse: Response | undefined

  globalThis.fetch = (async () => {
    streamingResponse = makeSseResponse(
      makeStreamChunks([
        {
          id: 'chatcmpl-1',
          object: 'chat.completion.chunk',
          model: 'fake-model',
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content: 'partial' },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chatcmpl-1',
          object: 'chat.completion.chunk',
          model: 'fake-model',
          choices: [
            {
              index: 0,
              delta: {},
              finish_reason: 'stop',
            },
          ],
        },
      ]),
    )

    return streamingResponse
  }) as FetchType

  const client = createOpenAIShimClient({}) as OpenAIShimClient
  const result = await client.beta.messages
    .create({
      model: 'fake-model',
      system: 'test system',
      messages: [{ role: 'user', content: 'hello' }],
      max_tokens: 64,
      stream: true,
    })
    .withResponse()

  for await (const event of result.data) {
    if (event.type === 'content_block_delta') {
      break
    }
  }

  expect(streamingResponse?.body?.locked).toBe(false)
})

test('does not emit null stop_reason when usage chunk arrives before finish chunk', async () => {
  globalThis.fetch = (async () => {
    const chunks = makeStreamChunks([
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: { role: 'assistant', content: 'hello world' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [],
        usage: {
          prompt_tokens: 123,
          completion_tokens: 45,
          total_tokens: 168,
        },
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'stop',
          },
        ],
      },
    ])

    return makeSseResponse(chunks)
  }) as FetchType

  const client = createOpenAIShimClient({}) as OpenAIShimClient
  const result = await client.beta.messages
    .create({
      model: 'fake-model',
      system: 'test system',
      messages: [{ role: 'user', content: 'hello' }],
      max_tokens: 64,
      stream: true,
    })
    .withResponse()

  const events = await collectEvents(result.data)
  const invalidStopReason = events.find(
    event =>
      event.type === 'message_delta' &&
      typeof event.delta === 'object' &&
      event.delta !== null &&
      (event.delta as Record<string, unknown>).stop_reason === null,
  )

  const usageEvent = events.find(
    event => event.type === 'message_delta' && typeof event.usage === 'object' && event.usage !== null,
  ) as { usage?: { input_tokens?: number; output_tokens?: number }; delta?: { stop_reason?: string } } | undefined

  expect(invalidStopReason).toBeUndefined()
  expect(usageEvent?.usage).toMatchObject({ input_tokens: 123, output_tokens: 45 })
  expect(usageEvent?.delta?.stop_reason).toBe('end_turn')
})

test('buffers tool argument deltas until the tool call is registered', async () => {
  globalThis.fetch = (async () => {
    const chunks = makeStreamChunks([
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: {
              role: 'assistant',
              tool_calls: [
                {
                  index: 1,
                  function: {
                    arguments: '{"command":',
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 1,
                  id: 'call_1',
                  type: 'function',
                  function: {
                    name: 'Bash',
                    arguments: '"pwd"}',
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'tool_calls',
          },
        ],
      },
    ])

    return makeSseResponse(chunks)
  }) as FetchType

  const client = createOpenAIShimClient({}) as OpenAIShimClient
  const result = await client.beta.messages
    .create({
      model: 'fake-model',
      system: 'test system',
      messages: [{ role: 'user', content: 'Use Bash' }],
      max_tokens: 64,
      stream: true,
    })
    .withResponse()

  const events = await collectEvents(result.data)
  const argumentDeltas = events
    .filter(
      event =>
        event.type === 'content_block_delta' &&
        typeof event.delta === 'object' &&
        event.delta !== null &&
        (event.delta as Record<string, unknown>).type === 'input_json_delta',
    )
    .map(event => (event.delta as Record<string, unknown>).partial_json)

  expect(argumentDeltas).toEqual(['{"command":', '"pwd"}'])
})

test('throws a helpful error when non-streaming OpenAI response is not valid JSON', async () => {
  globalThis.fetch = (async () =>
    new Response('<!DOCTYPE html><html><body>bad gateway</body></html>', {
      headers: {
        'Content-Type': 'text/html',
      },
    })) as FetchType

  const client = createOpenAIShimClient({}) as OpenAIShimClient

  await expect(
    client.beta.messages.create({
      model: 'fake-model',
      system: 'test system',
      messages: [{ role: 'user', content: 'hello' }],
      max_tokens: 64,
      stream: false,
    }),
  ).rejects.toThrow(/Failed to parse OpenAI response as JSON/)
})

test('normalizes assistant text blocks to string content in outgoing OpenAI messages', async () => {
  let requestBody: Record<string, unknown> | undefined

  globalThis.fetch = (async (_input, init) => {
    requestBody = JSON.parse(String(init?.body))

    return new Response(
      JSON.stringify({
        id: 'chatcmpl-1',
        model: 'fake-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: 'done',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 12,
          completion_tokens: 4,
          total_tokens: 16,
        },
      }),
      {
        headers: {
          'Content-Type': 'application/json',
        },
      },
    )
  }) as FetchType

  const client = createOpenAIShimClient({}) as OpenAIShimClient

  await client.beta.messages.create({
    model: 'fake-model',
    system: 'test system',
    messages: [
      { role: 'user', content: 'hello' },
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'first line' },
          { type: 'text', text: 'second line' },
        ],
      },
    ],
    max_tokens: 64,
    stream: false,
  })

  const assistantMessage = (requestBody?.messages as Array<Record<string, unknown>>).find(
    message => message.role === 'assistant',
  )

  expect(assistantMessage?.content).toBe('first linesecond line')
})
