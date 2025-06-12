import React, { useEffect, useState } from 'react'
import TodoItem from './components/TodoItem'
import AddTodo from './components/AddTodo'
import { getTodos, addTodo, updateTodo, deleteTodo } from './services/todo'
import { ITodo } from './types/todo'
import './App.css'

const App = () => {
  const [todos, setTodos] = useState<ITodo[]>([])

  useEffect(() => {
    fetchTodos()
  }, [])

  const fetchTodos = async () => {
    try {
      const {
        data: { todos }
      } = await getTodos()
      setTodos(todos)
    } catch (error) {
      console.log(`fetch todos error: ${error}`)
    }
  }

  const handleSaveTodo = (e: React.FormEvent, formData: ITodo): void => {
    e.preventDefault()
    addTodo(formData)
      .then(() => fetchTodos())
      .catch((err) => console.error(err))
  }

  const handleUpdateTodo = (todo: ITodo): void => {
    updateTodo(todo)
      .then(() => fetchTodos())
      .catch((err) => console.error(err))
  }

  const handleDeleteTodo = (id: string): void => {
    deleteTodo(id)
      .then(() => fetchTodos())
      .catch((err) => console.log(err))
  }

  return (
    <main className="App">
      <h1>My Todos</h1>
      <AddTodo saveTodo={handleSaveTodo} />
      {todos.map((todo: ITodo) => (
        <TodoItem key={todo.id} updateTodo={handleUpdateTodo} deleteTodo={handleDeleteTodo} todo={todo} />
      ))}
    </main>
  )
}

export default App
