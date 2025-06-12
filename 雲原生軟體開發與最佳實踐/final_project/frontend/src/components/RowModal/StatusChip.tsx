import { Order } from '@/types';
import { Chip } from '@nextui-org/chip';

export default function StatusChip({ order }: { order: Order }) {
  return (
    <Chip
      variant="flat"
      radius="sm"
      color={order.is_completed ? 'success' : 'primary'}
    >
      {order.is_completed ? '完成' : '進行中'}
    </Chip>
  );
}
