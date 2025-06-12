'use client';

import { useFormStatus } from 'react-dom';
import { Button } from '@nextui-org/react';


export default function SubmitButton() {
  const { pending } = useFormStatus();
    console.log(pending)
  return (
    <Button type="submit" disabled={pending} isLoading={pending}>
      登入
    </Button>
  );
}
