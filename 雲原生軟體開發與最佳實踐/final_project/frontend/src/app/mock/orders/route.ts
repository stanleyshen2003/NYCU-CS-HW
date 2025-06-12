import { orders } from '../data';

// eslint-disable-next-line import/prefer-default-export
export function GET() {
  return new Response(JSON.stringify({ data: orders }), { status: 200 });
}
