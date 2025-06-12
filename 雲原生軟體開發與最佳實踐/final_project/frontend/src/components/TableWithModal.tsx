'use client';

import { useDisclosure } from '@nextui-org/use-disclosure';
import { Key, useState } from 'react';
import { Order } from '@/types';
import RowModal from './RowModal';
import OrderTable from './OrderTable';
import Filter from './Filter';
import OrderCreator from './OrderCreator';

export type Action = 'admin-view' | 'admin-edit' | 'worker-view';

export default function TableWithModal({
  orders,
  action: defaultAction,
}: {
  orders: Order[];
  action: Action;
}) {
  const { isOpen, onOpen, onOpenChange, onClose } = useDisclosure();
  const [activeOrder, setActiveOrder] = useState<Order>();
  const [status, setStatus] = useState<boolean>();
  const [priority, setPriority] = useState<number>();
  const [action, setAction] = useState<Action>(defaultAction);

  const onRowAction = (id: Key) => {
    const order = orders.find((o) => o._id === id);
    setActiveOrder(order);
    onOpen();
  };

  const handleStatusChange = (newStatus: boolean | undefined) => {
    setStatus(newStatus);
  };

  const handlePriorityChange = (newPriority: number | undefined) => {
    setPriority(newPriority);
  };

  let filteredOrders = orders;

  if (status !== undefined) {
    filteredOrders = orders.filter((order) => status === order.is_completed);
  }

  if (priority !== undefined) {
    filteredOrders = filteredOrders.filter(
      (order) => order.priority === priority,
    );
  }

  return (
    <>
      <div className="flex justify-between">
        <Filter
          onStatusChange={handleStatusChange}
          onPriorityChange={handlePriorityChange}
        />
        {action !== 'worker-view' && <OrderCreator />}
      </div>
      <OrderTable
        orders={filteredOrders}
        onRowAction={onRowAction}
        action={action}
      />
      {activeOrder !== undefined && (
        <RowModal
          activeOrder={activeOrder}
          isOpen={isOpen}
          onOpenChange={onOpenChange}
          action={action}
          setAction={setAction}
          onClose={onClose}
        />
      )}
    </>
  );
}
