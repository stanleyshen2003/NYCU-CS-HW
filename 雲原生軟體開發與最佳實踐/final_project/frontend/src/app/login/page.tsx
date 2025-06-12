import SubmitButton from '@/components/SubmitButton';
import { Input } from '@nextui-org/react';
import { login } from './action';

export default async function LoginPage() {
  return (
    <div className="flex h-screen items-center justify-center bg-gray-100 ">
      <div className="w-[30rem] rounded-lg bg-white p-8 shadow-md">
        <div className="mb-10 flex items-center justify-center text-3xl font-bold ">
          <p>Login</p>
        </div>
        <form action={login} className="flex-col">
          <div className="mb-6 flex items-center justify-center ">
            <Input
              labelPlacement="outside"
              name="email"
              type="email"
              label="Email"
              variant="faded"
              placeholder="you@example.com"
              className="w-[80%]"
            />
          </div>
          <div className="mb-6 flex items-center justify-center">
            <Input
              labelPlacement="outside"
              name="password"
              type="password"
              label="Password"
              variant="faded"
              placeholder="Enter your password"
              className="w-[80%]"
            />
          </div>
          <div className="flex items-center justify-center">
            <SubmitButton />
          </div>
        </form>
      </div>
    </div>
  );
}
