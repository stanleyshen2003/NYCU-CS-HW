import mongoose from 'mongoose'

export const establishConnection = (connectionString: string) => mongoose.connect(connectionString)
