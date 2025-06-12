import { Order } from '@/types';
import { Chip } from '@nextui-org/chip';

export default function PriorityChip({ order }: { order: Order }) {
  return (
    <Chip
      variant="flat"
      radius="sm"
      color={
        order.priority === 1
          ? 'danger'
          : order.priority === 2
            ? 'warning'
            : 'default'
      }
    >
      {order.priority === 1 ? '特急單' : order.priority === 2 ? '急單' : '一般'}
    </Chip>
  );
}
