import { Key } from 'react';
import { Order } from '@/types';
import {
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
} from '@nextui-org/table';
import StatusChip from './RowModal/StatusChip';
import PriorityChip from './RowModal/PriorityChip';

type Action = 'admin-view' | 'admin-edit' | 'worker-view';

export default function OrderTable({
  orders,
  onRowAction,
  action,
}: {
  orders: Order[];
  onRowAction: (id: Key) => void;
  action: Action;
}) {
  return (
    <Table
      aria-label="Example static collection table"
      isHeaderSticky
      className="h-[70vh]"
      onRowAction={onRowAction}
      classNames={{
        base: 'w-[95vw] lg:w-[60vw]',
        tr: 'hover:bg-[#f4f4f5] transition-all cursor-pointer',
      }}
    >
      <TableHeader>
        <TableColumn>標題</TableColumn>
        <TableColumn>品管工程師</TableColumn>
        <TableColumn>Email</TableColumn>
        {action === 'worker-view' ? (
          <TableColumn>廠區</TableColumn>
        ) : (
          <TableColumn>實驗室</TableColumn>
        )}
        <TableColumn>優先序</TableColumn>
        <TableColumn>狀態</TableColumn>
      </TableHeader>
      <TableBody>
        {orders.map((order) => (
          <TableRow key={order._id} onClick={() => onRowAction(order._id)}>
            <TableCell>{order.title}</TableCell>
            <TableCell>{order.creator.split(' ')[1]}</TableCell>
            <TableCell>{order.creator.split(' ')[0]}</TableCell>
            {action === 'worker-view' ? (
              <TableCell>{order.fab_name}</TableCell>
            ) : (
              <TableCell>{order.lab_name}</TableCell>
            )}
            <TableCell>
              <PriorityChip order={order} />
            </TableCell>
            <TableCell>
              <StatusChip order={order} />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
