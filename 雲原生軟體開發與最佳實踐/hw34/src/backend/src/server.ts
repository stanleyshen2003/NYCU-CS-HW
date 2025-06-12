import fastify, { FastifyInstance, FastifyListenOptions } from 'fastify'
import { AppConfig } from './types/config'
import { establishConnection } from './plugins/mongodb'
import { TodoRouter } from './routes/todo'

export const serverOf: () => FastifyInstance = () => {
  const server = fastify()

  server.get('/ping', async (request, reply) => {
    return reply.status(200).send({ msg: 'pong!' })
  })

  server.register(TodoRouter, { prefix: '/api' })

  return server
}

export const serverStart: (appConfig: AppConfig) => (server: FastifyInstance) => Promise<FastifyInstance> =
  (appConfig) => async (server) => {
    await establishConnection(appConfig.mongoConnectionString)
    const listenOptions: FastifyListenOptions = {
      port: appConfig.port,
      host: appConfig.host
    }
    await server.listen(listenOptions)
    return server
  }
