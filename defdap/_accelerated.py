from numba import njit


@njit
def find_first(arr):
    for i in range(len(arr)):
        if arr[i]:
            return i


@njit
def flood_fill(seed, index, points_remaining, grains, boundary_x, boundary_y,
               added_coords):
    x, y = seed
    grains[y, x] = index
    points_remaining[y, x] = False
    edge = [seed]
    added_coords[0] = seed
    npoints = 1

    while edge:
        x, y = edge.pop()
        moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        # get rid of any that go out of the map area
        if x <= 0:
            moves.pop(1)
        elif x > grains.shape[1] - 2:
            moves.pop(0)
        if y <= 0:
            moves.pop(-1)
        elif y > grains.shape[0] - 2:
            moves.pop(-2)

        for (s, t) in moves:
            if grains[t, s] > 0:
                continue

            add_point = False

            if t == y:
                # moving horizontally
                if s > x:
                    # moving right
                    add_point = not boundary_x[y, x]
                else:
                    # moving left
                    add_point = not boundary_x[t, s]
            else:
                # moving vertically
                if t > y:
                    # moving down
                    add_point = not boundary_y[y, x]
                else:
                    # moving up
                    add_point = not boundary_y[t, s]

            if add_point:
                added_coords[npoints] = s, t
                grains[t, s] = index
                points_remaining[t, s] = False
                npoints += 1
                edge.append((s, t))

    return added_coords[:npoints]
