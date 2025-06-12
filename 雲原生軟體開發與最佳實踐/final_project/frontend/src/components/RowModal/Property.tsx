import { ReactNode } from 'react';

export default function Property({
  children,
  name,
}: {
  children: ReactNode;
  name: ReactNode;
}) {
  return (
    <div className="flex h-8 items-center">
      <div className="flex min-w-28 items-center gap-2">
        {name}
      </div>
      {children}
    </div>
  );
}
