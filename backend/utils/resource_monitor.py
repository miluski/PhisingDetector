import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import threading
from typing import Optional


class ResourceMonitor:

    def __init__(self, interval: float = 0.1, debug: bool = False):
        self.interval = interval
        self.debug = debug
        self.timestamps = []
        self.cpu_percent = []
        self.ram_gb = []
        self.start_time = None
        self.process = psutil.Process()
        self.num_cores = psutil.cpu_count()
        self._monitoring = False
        self._monitor_thread = None
        self._started = False

    def start(self):
        if self.debug:
            print(f"[INFO] ResourceMonitor uruchomiony z interwałem={self.interval}s")
        self._started = True
        self.start_time = time.time()
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        self._collect_sample()
        if self.debug:
            stats = self.get_stats()
            if stats:
                print(f"\n[INFO] Monitorowanie zakończone")
                print(f"[INFO] Czas trwania: {stats['duration']:.2f}s")
                print(f"[INFO] Zebrane próbki: {stats['samples']}")
                print(f"[INFO] CPU: {stats['cpu']['min']:.1f}%-{stats['cpu']['max']:.1f}% (średnia: {stats['cpu']['mean']:.1f}%)")
                print(f"[INFO] RAM: {stats['ram']['min']:.2f}GB-{stats['ram']['max']:.2f}GB (średnia: {stats['ram']['mean']:.2f}GB)")

    def _monitor_resources(self):
        while self._monitoring:
            self._collect_sample()
            time.sleep(self.interval)

    def _collect_sample(self):
        try:
            elapsed = time.time() - self.start_time
            cpu = self.process.cpu_percent(interval=None) / self.num_cores
            ram_bytes = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try:
                    ram_bytes += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            ram_gb = ram_bytes / (1024 ** 3)
            self.timestamps.append(elapsed)
            self.cpu_percent.append(cpu)
            self.ram_gb.append(ram_gb)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def plot(self, title: str = "Resource Usage", save_path: Optional[str] = None):
        if len(self.timestamps) == 0:
            print("Brak danych do wykresu")
            return

        times = np.array(self.timestamps)
        cpu_data = np.array(self.cpu_percent)
        ram_data = np.array(self.ram_gb)

        duration = times[-1]
        if duration < 180:
            times_display = times
            time_label = 'Czas (sekundy)'
        else:
            times_display = times / 60.0
            time_label = 'Czas (minuty)'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax1.plot(times_display, cpu_data, color='#2E86AB', linewidth=1.5, label='Użycie CPU')
        ax1.fill_between(times_display, cpu_data, alpha=0.3, color='#2E86AB')
        ax1.set_xlabel(time_label, fontsize=11, fontweight='bold')
        ax1.set_ylabel('Użycie CPU (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{title} - Użycie CPU', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, max(times_display) * 1.05)
        ax1.set_ylim(0, max(cpu_data.max() * 1.15, 10))
        ax1.legend(loc='upper right')
        cpu_stats = f'Min: {cpu_data.min():.1f}% | Max: {cpu_data.max():.1f}% | Śr: {cpu_data.mean():.1f}%'
        ax1.text(0.02, 0.98, cpu_stats, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.plot(times_display, ram_data, color='#A23B72', linewidth=1.5, label='Użycie RAM')
        ax2.fill_between(times_display, ram_data, alpha=0.3, color='#A23B72')
        ax2.set_xlabel(time_label, fontsize=11, fontweight='bold')
        ax2.set_ylabel('Użycie RAM (GB)', fontsize=11, fontweight='bold')
        ax2.set_title(f'{title} - Użycie RAM', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, max(times_display) * 1.05)
        ax2.set_ylim(0, max(ram_data.max() * 1.15, 0.5))
        ax2.legend(loc='upper right')
        ram_stats = f'Min: {ram_data.min():.2f}GB | Max: {ram_data.max():.2f}GB | Śr: {ram_data.mean():.2f}GB'
        ax2.text(0.02, 0.98, ram_stats, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            cpu_path = save_path.replace('.png', '_cpu.png')
            ram_path = save_path.replace('.png', '_ram.png')
            combined_path = save_path.replace('.png', '_combined.png')

            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            print(f"✓ Zapisano wykres łączony: {combined_path}")

            fig_cpu, ax_cpu = plt.subplots(1, 1, figsize=(12, 6))
            ax_cpu.plot(times_display, cpu_data, color='#2E86AB', linewidth=1.5)
            ax_cpu.fill_between(times_display, cpu_data, alpha=0.3, color='#2E86AB')
            ax_cpu.set_xlabel(time_label, fontsize=11, fontweight='bold')
            ax_cpu.set_ylabel('Użycie CPU (%)', fontsize=11, fontweight='bold')
            ax_cpu.set_title(f'{title} - Użycie CPU', fontsize=13, fontweight='bold')
            ax_cpu.grid(True, alpha=0.3, linestyle='--')
            ax_cpu.set_xlim(0, max(times_display) * 1.05)
            ax_cpu.set_ylim(0, max(cpu_data.max() * 1.15, 10))
            ax_cpu.text(0.02, 0.98, cpu_stats, transform=ax_cpu.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            plt.savefig(cpu_path, dpi=300, bbox_inches='tight')
            plt.close(fig_cpu)
            print(f"✓ Zapisano wykres CPU: {cpu_path}")

            fig_ram, ax_ram = plt.subplots(1, 1, figsize=(12, 6))
            ax_ram.plot(times_display, ram_data, color='#A23B72', linewidth=1.5)
            ax_ram.fill_between(times_display, ram_data, alpha=0.3, color='#A23B72')
            ax_ram.set_xlabel(time_label, fontsize=11, fontweight='bold')
            ax_ram.set_ylabel('Użycie RAM (GB)', fontsize=11, fontweight='bold')
            ax_ram.set_title(f'{title} - Użycie RAM', fontsize=13, fontweight='bold')
            ax_ram.grid(True, alpha=0.3, linestyle='--')
            ax_ram.set_xlim(0, max(times_display) * 1.05)
            ax_ram.set_ylim(0, max(ram_data.max() * 1.15, 0.5))
            ax_ram.text(0.02, 0.98, ram_stats, transform=ax_ram.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            plt.savefig(ram_path, dpi=300, bbox_inches='tight')
            plt.close(fig_ram)
            print(f"✓ Zapisano wykres RAM: {ram_path}")

        plt.show()

    def get_stats(self):
        if len(self.timestamps) == 0:
            return None
        cpu_data = np.array(self.cpu_percent)
        ram_data = np.array(self.ram_gb)
        return {
            'cpu': {
                'min': cpu_data.min(),
                'max': cpu_data.max(),
                'mean': cpu_data.mean(),
                'std': cpu_data.std()
            },
            'ram': {
                'min': ram_data.min(),
                'max': ram_data.max(),
                'mean': ram_data.mean(),
                'std': ram_data.std()
            },
            'duration': self.timestamps[-1] if self.timestamps else 0,
            'samples': len(self.timestamps)
        }
