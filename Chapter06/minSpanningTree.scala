import org.apache.spark.graphx._

def minSpanningTree[VD: scala.reflect.ClassTag](g: Graph[VD, Double]) = {
    // 初始化
    var g2 = g.mapEdges(e => (e.attr, false))
    // 遍历n-1条边, 最小生成树一共n-1条边
    (0L until g.vertices.count - 1).foreach { i: Long =>
        // 找到所有无用的边，即加上此边会与现有的生成树形成环
        val unvailableEdges = g2.outerJoinVertices(
            g2.subgraph(_.attr._2).connectedComponents.vertices
        )((vid, vd, cid) => (vd, cid)).subgraph(et =>
            et.srcAttr._2.getOrElse(-1) == et.dstAttr._2.getOrElse(-2)
        ).edges.map(e => ((e.srcId, e.dstId), e.attr))

        // 寻找权值最小的有效的边
        type edgeType = ((VertexId, VertexId), Double)
        val smallestEdge = g2.edges.map(e => ((e.srcId, e.dstId), e.attr)
            ).leftOuterJoin(unvailableEdges).filter(
            x => !x._2._1._2 && x._2._2.isEmpty
            ).map(x => (x._1, x._2._1._1)).min()(new Ordering[edgeType]() {
                override def compare(a: edgeType, b: edgeType) = {
                    val r = Ordering[Double].compare(a._2, b._2)
                    if (r == 0)
                        Ordering[Long].compare(a._1._1, b._1._1)
                    else
                        r
                }
            })
        println("iteration: " + i)
        println("smallestEdge: " + smallestEdge)
        // 加入最小生成树，变更状态
        g2 = g2.mapTriplets(et => (et.attr._1,
                et.attr._2 || (et.srcId == smallestEdge._1._1 &&
                et.dstId == smallestEdge._1._2)))
    }
    g2.subgraph(_.attr._2).mapEdges(_.attr._1)
}

val myVertices = sc.makeRDD(Array((1L, 'A'), (2L, 'B'), (3L, 'C'), (4L, 'D'),
    (5L, 'E'), (6L, "F"), (7L, "G")))
val myEdges = sc.makeRDD(Array(Edge(1L, 2L, 7.0), Edge(1L, 4L, 5.0),
    Edge(2L, 3L, 8.0), Edge(2L, 4L, 9.0), Edge(2L, 5L, 7.0), Edge(3L, 5L, 5.0),
    Edge(4L, 5L, 15.0), Edge(4L, 6L, 6.0), Edge(5L, 6L, 8.0),
    Edge(5L, 7L, 9.0), Edge(6L, 7L, 11.0)))
val myGraph = Graph(myVertices, myEdges)
minSpanningTree(myGraph).triplets.map(et => (et.srcAttr, et.dstAttr)).collect
