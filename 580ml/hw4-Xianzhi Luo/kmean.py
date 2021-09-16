import math
import numpy as np
centroids_set = [[2,3],[3,3],[5,4]]

points_set =[[1,1],[1,2],[2,1],[2,3],[3,3],[4,5],[5,4],[6,5]]
clusters_set = [[],[],[]]
pre_ce = []

def eu_distance(centroid,point):
	return math.sqrt((point[0]-centroid[0])**2+(point[1]-centroid[1])**2)


def min_dis(centroids,point):
	d = []
	for centroid in centroids:
		d.append(eu_distance(centroid,point))
	return (d.index(min(d)))	

def cluster(points,centroids,clusters):
	for point in points:
		clusters[min_dis(centroids,point)].append(point)
		



def sim_judge(pre_cen,cur_cen):
	if not pre_cen:
		return True
	for i in range(3):
		if((pre_cen[i][0] != cur_cen[i][0]) or (pre_cen[i][1]!=cur_cen[i][1])):
			return True
		else:
			return False
			
def get_centroids(clusters):
	mean_set =[]
	for cluster in clusters:
		np_clu = np.array(cluster)
		
		mean = (np.mean(np_clu,axis=0)).tolist()
		mean_set.append(mean)
	return mean_set

#centroids_set = get_centroids(clusters_set)


def sse_cal(clusters,centroids):
	sum = 0
	for i in range(3):
		for point in clusters[i]:
			sum += (eu_distance(centroids[i],point))**2
	return sum
		


initial = True
step = 1
while(sim_judge(pre_ce,centroids_set)):
	pre_ce = centroids_set
	cluster(points_set,centroids_set,clusters_set)
	centroids_set = get_centroids(clusters_set)
	sse = sse_cal(clusters_set,centroids_set)
	print "Step " + str(step)+":\n"
	print "The clusters formed: \n"
	for i in range(3):
		print "cluster " + str(i+1) +":"
		print clusters_set[i]
		print "\n"
	print "The new centroids: "
	print np.round(np.array(centroids_set),3).tolist()
	print "\n"
	print "The SSE: "
	print round(sse,3)
	print "\n"
	clusters_set=[[],[],[]]
	step += 1


