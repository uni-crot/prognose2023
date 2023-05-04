#function drawing snakes from gradient blocks
def our_snake(snake_block, snake_list, snake_color=[0,0,0]):
    i1=snake_color[0]
    i2=snake_color[1]
    i3=snake_color[2]
    for x in snake_list[::-1]:
        pygame.draw.rect(dis, (i1, i2, i3), [x[0], x[1], snake_block, snake_block])
        i1+=8
        i2+=8
        i3+=8
        if i1>200: i1 = 200
        if i2>200: i2 = 200
        if i3>200: i3 = 200