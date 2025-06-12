import { completeOrder, updateOrder } from '@/actions/order';
import { Order } from '@/types';
import { Button } from '@nextui-org/button';
import { useTransition } from 'react';

type Action = 'admin-view' | 'admin-edit' | 'worker-view';

export default function ActionButton({
  order,
  priority,
  action,
  setAction,
  onClose,
}: {
  order: Order;
  priority: number;
  action: Action;
  setAction: (action: Action) => void;
  onClose: () => void;
}) {
  const [isPending, startTransition] = useTransition();

  return (
    <Button
      radius="sm"
      className="bg-black text-white"
      isLoading={isPending}
      isDisabled={order.is_completed}
      onPress={() => {
        if (action === 'worker-view') {
          startTransition(async () => {
            await completeOrder(order._id);
            onClose();
          });
        }

        if (action === 'admin-view') {
          setAction('admin-edit');
        }

        if (action === 'admin-edit') {
          startTransition(async () => {
            await updateOrder(order._id, priority);
            setAction('admin-view');
            onClose();
          });
        }
      }}
    >
      {action === 'worker-view'
        ? '完成'
        : action === 'admin-view'
          ? '編輯'
          : '更新'}
    </Button>
  );
}
