import { NextResponse } from "next/server";
import clientPromise from "../../components/lib/mongodb";

export async function GET() {
  try {
    const client = await clientPromise;
    const db = client.db("test");
    const collection = db.collection("data_arca");

    const coolers = await collection
      .find({
        $or: [
          { temperature: { $lt: 1.57 } },
          { temperature: { $gt: 9.57 } },
        ],
      })
      .toArray();

    return NextResponse.json(coolers);
  } catch {
  return NextResponse.json({ error: "Error al obtener coolers" }, { status: 500 });
  }
}