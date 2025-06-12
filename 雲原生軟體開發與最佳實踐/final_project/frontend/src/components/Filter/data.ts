type Priority = {
  key: string;
  label: string;
  color: 'danger' | 'warning' | 'default';
};

export const priority: Priority[] = [
  { key: '1', label: '特急單', color: 'danger' },
  { key: '2', label: '急單', color: 'warning' },
  { key: '3', label: '一般', color: 'default' },
];

type Status = {
  key: number;
  label: string;
  color: 'primary' | 'success';
};

export const status: Status[] = [
  { key: 0, label: '進行中', color: 'primary' },
  { key: 1, label: '完成', color: 'success' },
];
