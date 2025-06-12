import { faker } from '@faker-js/faker';

const fabs = ['Fab A', 'Fab B', 'Fab C'];

const qaEngineers = Array.from({ length: 10 }, () => ({
  id: faker.string.uuid(),
  name: faker.person.fullName(),
  fab_name: faker.helpers.arrayElement(fabs),
}));

const labs = ['化學實驗室', '表面分析實驗室', '成分分析實驗室'];

export const labStaffs = Array.from({ length: 10 }, () => ({
  id: faker.string.uuid(),
  name: faker.person.fullName(),
  lab_name: faker.helpers.arrayElement(labs),
}));

export const orders = Array.from({ length: 20 }, () => ({
  id: faker.string.uuid(),
  title: faker.commerce.productName(),
  description: faker.commerce.productDescription(),
  creator: faker.helpers.arrayElement(qaEngineers).name,
  fab_name: faker.helpers.arrayElement(fabs),
  lab_name: faker.helpers.arrayElement(labs),
  priority: faker.number.int({ min: 1, max: 3 }),
  is_completed: faker.datatype.boolean(),
}));
