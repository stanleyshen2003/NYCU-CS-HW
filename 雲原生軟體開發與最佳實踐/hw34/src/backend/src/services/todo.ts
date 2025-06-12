import { ModifyResult } from 'mongoose'

import * as repo from '../repo/todo'
import { Todo, TodoBody } from '../types/todo'

export const getTodos: () => Promise<Array<Todo>> = async () => {
  const todos = await repo.findAllTodos()
  return todos
}

export const addTodo: (todoBody: TodoBody) => Promise<Todo> = async (todoBody) => {
  const newTodo = await repo.createTodo({
    ...todoBody
  })
  return newTodo
}

export const updateTodoStatus: (id: string, newStatus: boolean) => Promise<Todo | null> = async (id, newStatus) => {
  const todo = await repo.updateTodoById(id, { status: newStatus })
  return todo
}
export const deleteTodo: (id: string) => Promise<ModifyResult<Todo>> = async (id) => {
  const result = await repo.deleteTodoById(id)
  return result
}
