import org.apache.spark.graphx._
def dijkstra[VD](g: Graph[VD, Double], origin: VertexId): Graph[(VD, Double), Double] = {
    // 初始化
    var g2 = g.mapVertices{
        case (vid, _) => val vd = if (vid == origin) 0 else Double.MaxValue
        (false, vd)
    }
    // 遍历所有节点
    (0L until g.vertices.count).foreach { i: Long =>
        // 确定最短路径值最小的作为当前顶点
        val currentVertexId = g2.vertices.filter(!_._2._1).
                fold((0L, (false, Double.MaxValue))) {
                    case (a, b) => if (a._2._2 < b._2._2) a else b
                }._1
        // 更新当前顶点关联的所有节点最短路径
        val newDistances: VertexRDD[Double] = g2.aggregateMessages(ctx => if (ctx.srcId == currentVertexId)
                ctx.sendToDst(ctx.attr + ctx.srcAttr._2),
                (a, b) => if (a >= b) b else a
                )
        // 更新当前顶点为已访问点
        g2 = g2.outerJoinVertices(newDistances) { (vid, vd, newSum) =>
                (vd._1 || vid == currentVertexId,
                        math.min(vd._2, newSum.getOrElse(Double.MaxValue)))
        }
    }
    g.outerJoinVertices(g2.vertices) { (vid, vd, dist) =>
            (vd, dist.getOrElse((false, Double.MaxValue))._2)
    }
}
val myVertices = sc.makeRDD(Array((1L, 'A'), (2L, 'B'), (3L, 'C'), (4L, 'D'),
    (5L, 'E'), (6L, "F"), (7L, "G")))
val myEdges = sc.makeRDD(Array(Edge(1L, 2L, 7.0), Edge(1L, 4L, 5.0),
    Edge(2L, 3L, 8.0), Edge(2L, 4L, 9.0), Edge(2L, 5L, 7.0), Edge(3L, 5L, 5.0),
    Edge(4L, 5L, 15.0), Edge(4L, 6L, 6.0), Edge(5L, 6L, 8.0),
    Edge(5L, 7L, 9.0), Edge(6L, 7L, 11.0)))
val myGraph = Graph(myVertices, myEdges)
dijkstra(myGraph, 1L).vertices.map(_._2).collect
