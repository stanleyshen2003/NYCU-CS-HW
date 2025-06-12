import { Link } from '@nextui-org/react';

export default async function Home() {
  return (
    <div className="flex h-screen flex-col items-center justify-center p-10">
      <Link href="/admin">Go to Admin Page</Link>
      <Link href="/worker">Go to Worker Page</Link>
    </div>
  );
}
