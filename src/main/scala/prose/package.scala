import chisel3._

package object prose {
  def splitToChunks(a: UInt, width: Int): Vec[UInt] =
    VecInit((0 until a.getWidth / width).map(idx => a((idx + 1) * width - 1, idx * width)).reverse)

  def assign[T <: Data](dst: T, src: T): Unit = dst := src
  def assign[T <: Data](a: (T, T)): Unit = a._1 := a._2
}
