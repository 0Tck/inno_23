import { Module } from '@nestjs/common';
import { SequelizeModule } from '@nestjs/sequelize';

import { Sample } from './sample.model';
import { SampleService } from './sample.service';
import { SampleController } from './sample.controller';

@Module({
  imports: [SequelizeModule.forFeature([Sample])],
  providers: [SampleService],
  controllers: [SampleController],
})
export class SampleModule {}
