from train import Agent


if __name__ == '__main__':
    agent = Agent(mode='test', model_root='logs/epochs_10_lr_2e-05_batch_32_google-bert_fix_full_no_video/epoch_9.pth', batch_size=6)
    answers = agent.inference()
    with open('submission5.csv', 'w') as f:
        f.write('index,rating\n')
        for i, answer in enumerate(answers):
            answer = answer+1
            f.write('index_'+str(i)+','+str(answer)+'\n')