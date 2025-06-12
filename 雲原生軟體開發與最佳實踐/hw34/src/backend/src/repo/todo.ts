import { ModifyResult, UpdateQuery } from 'mongoose'

import TodoModel from '../models/todo'
import { Todo, TodoBody } from '../types/todo'

export const findAllTodos: () => Promise<Array<Todo>> = () => TodoModel.find().exec()

export const createTodo: (todoBody: TodoBody) => Promise<Todo> = (todoBody) => TodoModel.create(todoBody)

export const updateTodoById: (id: string, update: UpdateQuery<TodoBody>) => Promise<Todo | null> = (id, update) =>
  TodoModel.findByIdAndUpdate(id, update, { new: true })

export const deleteTodoById: (id: string) => Promise<ModifyResult<Todo>> = (id) =>
  TodoModel.findByIdAndDelete(id).exec()
