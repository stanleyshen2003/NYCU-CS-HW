'use client';

import { Select, SelectItem } from '@nextui-org/react';
import { priority, status } from './data';

function Filter({
  onStatusChange,
  onPriorityChange,
}: {
  onStatusChange: (status: boolean | undefined) => void;
  onPriorityChange: (priority: number | undefined) => void;
}) {
  return (
    <div className="flex w-full flex-wrap gap-4 md:flex-nowrap">
      <Select
        size="sm"
        label="優先序"
        className="max-w-[10rem]"
        variant="faded"
        onChange={(e) => {
          if (e.target.value === '') {
            onPriorityChange(undefined);
            return;
          }

          onPriorityChange(Number(e.target.value));
        }}
      >
        {priority.map((p) => (
          <SelectItem key={p.key} variant="flat" color={p.color}>
            {p.label}
          </SelectItem>
        ))}
      </Select>
      <Select
        size="sm"
        label="狀態"
        className="max-w-[10rem]"
        variant="faded"
        onChange={(e) => {
          if (e.target.value === '') {
            onStatusChange(undefined);
            return;
          }

          onStatusChange(e.target.value === '1');
        }}
      >
        {status.map((s) => (
          <SelectItem key={s.key} variant="flat" color={s.color}>
            {s.label}
          </SelectItem>
        ))}
      </Select>
    </div>
  );
}

export default Filter;
