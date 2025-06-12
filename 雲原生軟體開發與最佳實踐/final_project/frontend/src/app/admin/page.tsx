import TableWithModal from '@/components/TableWithModal';
import AvatarButton from '@/components/Avatar';
import { Order } from '@/types';
import { cookies } from 'next/headers';

export default async function AdminPage() {
  const accessToken = cookies().get('accessToken')!.value;
  const res = await fetch(`${process.env.API_URL}/orders`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
  const data: Order[] = await res.json();

  const name = cookies().get('name')!.value;
  const position = cookies().get('departmentName')!.value;

  console.log(data);
  return (
    <div className="flex h-screen flex-col items-center justify-center">
      <div className="flex flex-col gap-5">
        <div className="flex items-center justify-between">
          <div className="text-3xl font-bold">委託單列表</div>
          <div className="flex gap-5">
            <AvatarButton name={name} position={position} />
          </div>
        </div>
        <TableWithModal orders={data} action="admin-view" />
      </div>
    </div>
  );
}
