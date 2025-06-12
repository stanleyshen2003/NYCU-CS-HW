'use server';

import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

export async function login(formData: FormData) {
  try {
    const email = formData.get('email') as string;
    const password = formData.get('password') as string;
    const response = await fetch(`${process.env.API_URL}/staffs/signin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }
    const result = await response.json();
    cookies().set('accessToken', result.token);
    cookies().set('position', result.position);
    cookies().set('name', result.name);
    cookies().set('departmentName', result.department_name);
  } catch (error) {
    console.error('Error during user validation:', error);
    return;
  }

  redirect('/');
}

export async function logOut() {
  cookies().delete('accessToken');
  cookies().delete('position');
  cookies().delete('name');
  cookies().delete('departmentName');
  redirect('/login');
}
