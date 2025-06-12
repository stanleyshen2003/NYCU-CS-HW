import { FastifyInstance, RouteShorthandOptions } from 'fastify'

import { addTodo, deleteTodo, getTodos, updateTodoStatus } from '../services/todo'
import { TodoBody } from '../types/todo'

export const TodoRouter = (server: FastifyInstance, opts: RouteShorthandOptions, done: (error?: Error) => void) => {
  interface IdParam {
    id: string
  }
  interface StatusBody {
    status: boolean
  }

  server.get('/v1/todos', async (request, reply) => {
    try {
      const todos = await getTodos()
      return reply.status(200).send({ todos })
    } catch (error) {
      server.log.error(`GET /v1/todos Error: ${error}`)
      return reply.status(500).send(`[Server Error]: ${error}`)
    }
  })

  server.post<{ Body: TodoBody }>('/v1/todos', async (request, reply) => {
    try {
      const todoBody = request.body
      const todo = await addTodo(todoBody)
      return reply.status(201).send({ todo })
    } catch (error) {
      server.log.error(`POST /v1/todos Error: ${error}`)
      return reply.status(500).send(`[Server Error]: ${error}`)
    }
  })

  server.put<{ Params: IdParam; Body: StatusBody }>('/v1/todos/:id', opts, async (request, reply) => {
    try {
      const id = request.params.id
      const status = request.body.status
      const todo = await updateTodoStatus(id, status)
      if (todo) {
        return reply.status(200).send({ todo })
      } else {
        return reply.status(404).send({ msg: `Not Found Todo:${id}` })
      }
    } catch (error) {
      server.log.error(`PUT /v1/todos/${request.params.id} Error: ${error}`)
      return reply.status(500).send(`[Server Error]: ${error}`)
    }
  })

  server.delete<{ Params: IdParam }>('/v1/todos/:id', opts, async (request, reply) => {
    try {
      const id = request.params.id
      const todo = await deleteTodo(id)
      if (todo) {
        return reply.status(204).send()
      } else {
        return reply.status(404).send({ msg: `Not Found Todo:${id}` })
      }
    } catch (error) {
      server.log.error(`DELETE /v1/todos/${request.params.id} Error: ${error}`)
      return reply.status(500).send(`[Server Error]: ${error}`)
    }
  })

  done()
}
