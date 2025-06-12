export interface ITodo {
  id: string
  name: string
  description: string
  status: boolean
  createdAt?: string
  updatedAt?: string
}

export interface TodoProps {
  todo: ITodo
}

export type TodoResponse = {
  todos: Array<ITodo>
}
