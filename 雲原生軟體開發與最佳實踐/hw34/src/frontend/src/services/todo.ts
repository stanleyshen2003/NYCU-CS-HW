import axios, { AxiosResponse } from 'axios'
import { ITodo, TodoResponse } from '../types/todo'

export const getTodos = async (): Promise<AxiosResponse<TodoResponse>> => {
  try {
    const res = await axios.get<TodoResponse>(`/api/v1/todos`)
    return Promise.resolve(res)
  } catch (error) {
    return Promise.reject(`GET /todos ERROR: ${error}`)
  }
}

export const addTodo = async (todoBody: ITodo): Promise<AxiosResponse<ITodo>> => {
  try {
    const newTodo = {
      ...todoBody,
      status: false
    }
    const todo = await axios.post(`/api/v1/todos`, newTodo)
    return todo
  } catch (error) {
    console.error(`POST /api/v1/todos ERROR: ${error}`)
    throw new Error(`${error}`)
  }
}

export const updateTodo = async (todoBody: ITodo): Promise<AxiosResponse<ITodo>> => {
  try {
    const payload = {
      status: true
    }
    const todo = await axios.put(`/api/v1/todos/${todoBody.id}`, payload)
    return todo
  } catch (error) {
    console.error(`PUT /api/v1/todos/${todoBody.id} ERROR: ${error}`)
    throw new Error(`${error}`)
  }
}

export const deleteTodo = async (id: string): Promise<AxiosResponse> => {
  try {
    const res = await axios.delete(`/api/v1/todos/${id}`)
    return res
  } catch (error) {
    console.error(`DELETE /api/v1/todos/${id} ERROR: ${error}`)
    throw new Error(`${error}`)
  }
}
