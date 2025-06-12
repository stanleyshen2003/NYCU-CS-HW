'use server';

import { revalidatePath } from 'next/cache';
import { cookies } from 'next/headers';

export async function createOrder(
  formData: FormData
) {
  const accessToken = cookies().get('accessToken')!.value

  const res = await fetch(`${process.env.API_URL}/orders`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
    body: formData,
  });

  if (!res.ok) {
    const errorText = await res.text();
    console.error(`Failed to create order. Status: ${res.status}. Response: ${errorText}`);
    throw new Error(`Failed to create order. Status: ${res.status}. Response: ${errorText}`);
  }

  revalidatePath('/');
}

export async function updateOrder(id: string, priority: number) {
  const accessToken = cookies().get('accessToken')!.value;
  const formData = new FormData();
  formData.append('_id', id);
  formData.append('priority', priority.toString());

  const res = await fetch(`${process.env.API_URL}/orders`, {
    method: 'PUT',
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
    body: formData,
  });

  if (!res.ok) {
    throw new Error('Failed to create order');
  }

  revalidatePath('/');
}

export async function completeOrder(id: string) {
  const accessToken = cookies().get('accessToken')!.value;

  const res = await fetch(`${process.env.API_URL}/orders/${id}`, {
    method: 'PUT',
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });

  if (!res.ok) {
    console.error('Failed to complete order');
    return;
  }

  revalidatePath('/');
}
