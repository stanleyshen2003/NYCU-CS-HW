import TableWithModal from '@/components/TableWithModal';
import { Order } from '@/types';
import AvatarButton from '@/components/Avatar';
import { cookies } from 'next/headers';

export default async function WorkerPage() {
  const accessToken = cookies().get('accessToken')!.value;
  const res = await fetch(`${process.env.API_URL}/orders`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
  const data: Order[] = await res.json();

  const name = cookies().get('name')!.value;
  const position = cookies().get('departmentName')!.value;

  return (
    <div className="flex h-screen flex-col items-center justify-center">
      <div className="flex flex-col gap-5">
        <div className="flex items-center justify-between">
          <div className="text-3xl font-bold">委託單列表</div>
          <AvatarButton name={name} position={position} />
        </div>
        <TableWithModal orders={data} action="worker-view" />
      </div>
    </div>
  );
}
