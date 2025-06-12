'use client';

import {
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
  User,
} from '@nextui-org/react';
import { AvatarButtonProps } from '@/types';
import { logOut } from '@/app/login/action';

export default function AvatarButton({ name, position }: AvatarButtonProps) {
  return (
    <div className="flex items-center gap-4">
      <Dropdown placement="bottom-start">
        <DropdownTrigger>
          <User
            as="button"
            avatarProps={{
              isBordered: true,
              src: 'https://www.svgrepo.com/show/418965/user-avatar-profile.svg',
            }}
            className="transition-transform"
            description={position}
            name={name}
          />
        </DropdownTrigger>
        <DropdownMenu aria-label="User Actions" variant="flat">
          <DropdownItem
            key="logout"
            color="danger"
            onPress={async () => {
              await logOut();
            }}
          >
            Log Out
          </DropdownItem>
        </DropdownMenu>
      </Dropdown>
    </div>
  );
}
