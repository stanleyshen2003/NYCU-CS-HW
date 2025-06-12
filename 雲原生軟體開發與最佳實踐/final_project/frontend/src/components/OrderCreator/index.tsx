'use client';

import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
} from '@nextui-org/modal';
import { Button } from '@nextui-org/button';
import { useDisclosure } from '@nextui-org/use-disclosure';
import {
  Divider,
  Input,
  Select,
  SelectItem,
  Textarea,
} from '@nextui-org/react';
import { useState} from 'react';
import { createOrder } from '@/actions/order';
import { Lab, Priority, Upload } from '../Icons';
import Property from '../RowModal/Property';
import SubmitButton from './SubmitButton';

export default function OrderCreator() {
  const { isOpen, onOpen, onOpenChange, onClose } = useDisclosure();
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [lab, setLab] = useState('化學實驗室');
  const [priority, setPriority] = useState(3);
  const [file, setFile] = useState<File | null>(null);


  const handleSubmit = async () => {
    onClose();
    setTitle('');
    setDescription('');
    setLab('化學實驗室');
    setPriority(3);
    setFile(null);
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files ? e.target.files[0] : null;
    setFile(selectedFile);
  };
  return (
    <>
      <Button
        radius="sm"
        size="lg"
        variant="faded"
        className="text-sm"
        onPress={onOpen}
      >
        新增委託單
      </Button>
      <Modal
        size="lg"
        hideCloseButton
        isOpen={isOpen}
        onOpenChange={onOpenChange}
      >
        <ModalContent>
          <form action={createOrder} onSubmit={handleSubmit}>
            <ModalHeader className="text-xl">
              <Input
                name ="title"
                aria-label="委託單名稱"
                variant="faded"
                size="lg"
                radius="sm"
                placeholder="委託單名稱"
                value={title}
                onValueChange={setTitle}
              />
            </ModalHeader>
            <ModalBody>
              <div className="flex flex-col gap-2 text-sm">
                <Property
                  name={
                    <>
                      <Lab />
                      實驗室
                    </>
                  }
                >
                  <Select
                    name='lab_name'
                    aria-label="實驗室"
                    variant="faded"
                    size="sm"
                    radius="sm"
                    selectedKeys={[lab]}
                    onChange={(e) => {
                      if (e.target.value === '') {
                        return;
                      }

                      setLab(e.target.value);
                    }}
                  >
                    <SelectItem key="化學實驗室">化學實驗室</SelectItem>
                    <SelectItem key="表面分析實驗室">表面分析實驗室</SelectItem>
                    <SelectItem key="成分分析實驗室">成分分析實驗室</SelectItem>
                  </Select>
                </Property>
                <Property
                  name={
                    <>
                      <Priority />
                      優先序
                    </>
                  }
                >
                  <Select
                    name='priority'
                    aria-label="優先序"
                    variant="faded"
                    size="sm"
                    radius="sm"
                    onChange={(e) => {
                      setPriority(Number(e.target.value));
                    }}
                    selectedKeys={[String(priority)]}
                  >
                    <SelectItem
                      variant="faded"
                      color="danger"
                      key="1"
                      value={1}
                    >
                      特急單
                    </SelectItem>
                    <SelectItem
                      variant="faded"
                      color="warning"
                      key="2"
                      value={2}
                    >
                      急單
                    </SelectItem>
                    <SelectItem
                      variant="faded"
                      color="default"
                      key="3"
                      value={3}
                    >
                      一般
                    </SelectItem>
                  </Select>
                </Property>
              </div>
              <Divider />
              <Textarea
                name='description'
                aria-label="委託單內容"
                variant="faded"
                radius="sm"
                placeholder="委託單內容"
                value={description}
                onValueChange={setDescription}
              />
              <Divider />
              <div className="flex items-center gap-4">
                <Button
                  radius="sm"
                  size="lg"
                  variant="bordered"
                  onPress={() => {
                    document.getElementById('file-input')?.click();
                  }}
                  className="flex items-center gap-2 rounded border border-gray-300 bg-gray-100 px-4 py-2"
                >
                  <Upload />
                  <span>選擇檔案</span>
                </Button>
                <input
                  name='file'
                  id="file-input"
                  type="file"
                  onChange={handleFileChange}
                  accept=".pdf"
                  className="hidden"
                />
                <span>{file ? file.name : '未選擇任何檔案'}</span>
              </div>
            </ModalBody>
            <ModalFooter>
              <SubmitButton />
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
    </>
  );
}
