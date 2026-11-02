import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "ChronoRig | Autonomous Video-to-Physics 3D Studio",
    description:
        "Convert standard videos of moving mechanisms into verified Three.js/Rapier and Blender (bpy) 3D physics rigs with Meta SAM 3.1 and Muse Spark.",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" className="dark">
            <body className="bg-slate-950 text-slate-100 antialiased selection:bg-teal-500 selection:text-slate-950">
                {children}
            </body>
        </html>
    );
}