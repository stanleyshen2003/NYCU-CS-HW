import React, { useEffect, useState } from 'react'
import { ITodo } from '../types/todo'

const AddTodo = (props: { saveTodo: (e: React.FormEvent, formData: ITodo) => void }) => {
  const [formData, setFormData] = useState<ITodo>({
    id: '',
    name: '',
    description: '',
    status: false
  })
  const [isDisabled, setIsDisabled] = useState<boolean>(() => false)

  const handleForm = (e: React.FormEvent<HTMLInputElement>): void => {
    setFormData({
      ...formData,
      [e.currentTarget.id]: e.currentTarget.value
    })
  }

  useEffect(() => {
    setIsDisabled(formData.name === '' || formData.description === '')
  }, [formData])

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    props.saveTodo(e, formData)
    setFormData({ id: '', name: '', description: '', status: false })
  }

  return (
    <form className="Form" onSubmit={(e) => handleSubmit(e)}>
      <div>
        <div>
          <label htmlFor="name">Name</label>
          <input onChange={handleForm} type="text" id="name" value={formData.name} required />
        </div>
        <div>
          <label htmlFor="description">Description</label>
          <input onChange={handleForm} type="text" id="description" value={formData.description} required />
        </div>
      </div>
      <button disabled={isDisabled}>Add Todo</button>
    </form>
  )
}

export default AddTodo
