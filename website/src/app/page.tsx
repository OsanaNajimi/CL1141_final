import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import About from '@/components/About';
import Architecture from '@/components/Architecture';

export default function Home() {
  return (
    <main style={{ minHeight: '100vh', backgroundColor: '#111' }}>
      <Navbar />
      <Hero />
      <About />
      <Architecture />
    </main>
  );
}
