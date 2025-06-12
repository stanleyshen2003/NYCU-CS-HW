export type Fab = 'Fab A' | 'Fab B' | 'Fab C';

export type Lab = '化學實驗室' | '表面分析實驗室' | '成分分析實驗室';

export type QAEngineer = {
  id: string;
  name: string;
  fab_name: Fab;
};

export type LabStaff = {
  id: string;
  name: string;
  lab_name: Lab;
};

export type Order = {
  attachments: any;
  _id: string;
  title: string;
  description: string;
  creator: string;
  fab_name: Fab;
  lab_name: Lab;
  priority: number;
  is_completed: boolean;
};

export type FilterProps = {
  onStatusChange: (status: number | undefined) => void;
  onPriorityChange: (priority: number | undefined) => void;
};

export type AvatarButtonProps={
  name: string;
  position: string;
}

