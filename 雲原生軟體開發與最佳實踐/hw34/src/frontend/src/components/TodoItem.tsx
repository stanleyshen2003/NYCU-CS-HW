import { ITodo } from '../types/todo'

const Todo = (props: { todo: ITodo; updateTodo: (todo: ITodo) => void; deleteTodo: (id: string) => void }) => {
  const checkTodo: string = props.todo.status ? `line-through` : ''

  return (
    <div className="Card">
      <div className="Card--text">
        <h1 className={checkTodo}>{props.todo.name}</h1>
        <span className={checkTodo}>{props.todo.description}</span>
      </div>
      <div className="Card--button">
        <button
          onClick={() => props.updateTodo(props.todo)}
          className={props.todo.status ? `hide-button` : `Card--button__done`}
        >
          Complete
        </button>
        <button onClick={() => props.deleteTodo(props.todo.id)} className="Card-button__delete">
          Delete
        </button>
      </div>
    </div>
  )
}

export default Todo
