'use client';

import { useFormStatus } from 'react-dom';
import { Button } from '@nextui-org/react';

export default function SubmitButton() {
  const { pending } = useFormStatus();
  console.log(pending);
  return (
    <Button
      radius="sm"
      className="bg-black text-white"
      type="submit"
      disabled={pending}
      isLoading={pending}
    >
      新增
    </Button>
  );
}
