export type Todo = {
  id: string
  name: string
  description: string
  status: boolean
}

export type TodoBody = Omit<Todo, 'id'>
