import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  const accessToken = request.cookies.get('accessToken');
  const position = request.cookies.get('position');

  if (accessToken === undefined && request.nextUrl.pathname !== '/login') {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  if (accessToken !== undefined && request.nextUrl.pathname === '/login') {
    return NextResponse.redirect(new URL('/', request.url));
  }

  if (
    request.nextUrl.pathname === '/' ||
    request.nextUrl.pathname === '/admin' ||
    request.nextUrl.pathname === '/worker'
  ) {
    if (position?.value === 'Fab') {
      return NextResponse.rewrite(new URL('/admin', request.url));
    }
    return NextResponse.rewrite(new URL('/worker', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|.*\\..*|favicon.ico|mock).*)'],
};
