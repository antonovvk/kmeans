package kmeans

import (
	"fmt"
	"math"
	"math/rand"
)

// Observation: Data Abstraction for an N-dimensional
// observation
type Observation []float64

// Abstracts the Observation with a cluster number
// Update and computeation becomes more efficient
type ClusteredObservation struct {
	ClusterNumber int
	Distance      float64
	Observation
}

// Distance Function: To compute the distanfe between observations
type DistanceFunction func(first, second []float64) (float64, error)

/*
func (observation Observation) Sqd(otherObservation Observation) (ssq float64) {
	for ii, jj := range observation {
		d := jj - otherObservation[ii]
		ssq += d * d
	}
	return ssq
}
*/

// Summation of two vectors
func (observation Observation) Add(otherObservation Observation) {
	for ii, jj := range otherObservation {
		observation[ii] += jj
	}
}

// Multiplication of a vector with a scalar
func (observation Observation) Mul(scalar float64) {
	for ii := range observation {
		observation[ii] *= scalar
	}
}

// Dot Product of Two vectors
func (observation Observation) InnerProduct(otherObservation Observation) {
	for ii := range observation {
		observation[ii] *= otherObservation[ii]
	}
}

// Outer Product of two arrays
// TODO: Need to be tested
func (observation Observation) OuterProduct(otherObservation Observation) [][]float64 {
	result := make([][]float64, len(observation))
	for ii := range result {
		result[ii] = make([]float64, len(otherObservation))
	}
	for ii := range result {
		for jj := range result[ii] {
			result[ii][jj] = observation[ii] * otherObservation[jj]
		}
	}
	return result
}

// Find the closest observation and return the distance
// Index of observation, distance
func (p ClusteredObservation) near(mean []Observation, distanceFunction DistanceFunction) (int, float64) {
	indexOfCluster := 0
	minSquaredDistance, _ := distanceFunction(p.Observation, mean[0])
	for i := 1; i < len(mean); i++ {
		squaredDistance, _ := distanceFunction(p.Observation, mean[i])
		if squaredDistance < minSquaredDistance {
			minSquaredDistance = squaredDistance
			indexOfCluster = i
		}
	}
	return indexOfCluster, math.Sqrt(minSquaredDistance)
}

func (p *ClusteredObservation) update(mean []Observation, distanceFunction DistanceFunction) (bool, error) {
	cc, dist := p.near(mean, distanceFunction)
	if math.IsNaN(dist) {
		return false, fmt.Errorf("Distance is NaN")
	}
	upd := p.ClusterNumber != cc
	p.ClusterNumber = cc
	p.Distance = dist
	return upd, nil
}

// Instead of initializing randomly the seeds, make a sound decision of initializing
func seed(data []ClusteredObservation, k int, distanceFunction DistanceFunction) []Observation {
	s := make([]Observation, k)
	s[0] = data[rand.Intn(len(data))].Observation
	d2 := make([]float64, len(data))
	for ii := 1; ii < k; ii++ {
		var sum float64
		for jj, p := range data {
			_, dMin := p.near(s[:ii], distanceFunction)
			d2[jj] = dMin * dMin
			sum += d2[jj]
		}
		target := rand.Float64() * sum
		jj := 0
		for sum = d2[0]; sum < target; sum += d2[jj] {
			jj++
		}
		s[ii] = data[jj].Observation
	}
	return s
}

// K-Means Algorithm
func kmeans(data []ClusteredObservation, mean []Observation, distanceFunction DistanceFunction, threshold int) ([]ClusteredObservation, float64, error) {
	var sumDist float64
	for ii := range data {
		if _, err := data[ii].update(mean, distanceFunction); err != nil {
			return nil, 0, err
		}
		sumDist += data[ii].Distance
	}

	mLen := make([]int, len(mean))
	n := len(data[0].Observation)
	counter := 0
	for step := 0; ; step++ {
		sumDist = 0
		for ii := range mean {
			mean[ii] = make(Observation, n)
			mLen[ii] = 0
		}
		for _, p := range data {
			mean[p.ClusterNumber].Add(p.Observation)
			mLen[p.ClusterNumber]++
		}
		for ii := range mean {
			mean[ii].Mul(1 / float64(mLen[ii]))
		}
		var changes int
		for ii := range data {
			if upd, err := data[ii].update(mean, distanceFunction); err != nil {
				return nil, 0, err
			} else if upd {
				changes++
			}
			sumDist += data[ii].Distance
		}
		counter++
		if changes == 0 || counter > threshold {
			break
		}
	}
	return data, sumDist, nil
}

// K-Means Algorithm with smart seeds
// as known as K-Means ++
func Kmeans(rawData [][]float64, k int, distanceFunction DistanceFunction, threshold int) ([]int, error) {
	data := make([]ClusteredObservation, len(rawData))
	for ii, jj := range rawData {
		data[ii].Observation = jj
	}
	seeds := seed(data, k, distanceFunction)
	clusteredData, _, err := kmeans(data, seeds, distanceFunction, threshold)
	labels := make([]int, len(clusteredData))
	for ii, jj := range clusteredData {
		labels[ii] = jj.ClusterNumber
	}
	return labels, err
}

func RepeatedKmeans(rawData [][]float64, k, r int, distanceFunction DistanceFunction, threshold int) ([]int, error) {
	bestRun := -1
	bestDist := float64(-1)
	labels := make([][]int, r)
	for run := 0; run < r; run++ {
		// Just for fun
		rand.Seed(int64(run))

		data := make([]ClusteredObservation, len(rawData))
		for ii, jj := range rawData {
			data[ii].Observation = jj
		}
		seeds := seed(data, k, distanceFunction)
		clusteredData, sumDist, err := kmeans(data, seeds, distanceFunction, threshold)
		if err != nil {
			return nil, err
		}
		labels[run] = make([]int, len(clusteredData))
		for ii, jj := range clusteredData {
			labels[run][ii] = jj.ClusterNumber
		}
		if bestDist == -1 || sumDist < bestDist {
			bestDist = sumDist
			bestRun = run
		}
	}
	return labels[bestRun], nil
}
