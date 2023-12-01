import { InjectModel } from '@nestjs/sequelize';
import { Injectable, HttpException, HttpStatus } from '@nestjs/common';

import { Sample } from './sample.model';
import { CreateSampleDto, UpdateSampleDto } from './dto';

@Injectable()
export class SampleService {
  constructor(
    @InjectModel(Sample)
    private sampleModel: typeof Sample,
  ) {}

  async createSample(payload: CreateSampleDto): Promise<Sample> {
    try {
      const queryResponse = await this.sampleModel.create(payload);
      return queryResponse;
    } catch (error) {
      console.error(error);
      throw new HttpException(
        `Error in Creating New Sample - ${error.name}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async getAll(): Promise<Sample[]> {
    try {
      const queryResponse = await this.sampleModel.findAll();
      return queryResponse;
    } catch (error) {
      console.error(error);
      throw new HttpException(
        `Error in Getting All Samples - ${error.name}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async getById(sampleId: number): Promise<Sample> {
    try {
      const queryResponse = await this.sampleModel.findOne({
        where: {
          id: sampleId,
        },
      });
      return queryResponse;
    } catch (error) {
      console.error(error);
      throw new HttpException(
        `Error in Getting Sample with ID '${sampleId}' - ${error.name}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async updateSample(
    sampleId: number,
    payload: UpdateSampleDto,
  ): Promise<[affectedCount: number]> {
    try {
      const queryResponse = await this.sampleModel.update(payload, {
        where: {
          id: sampleId,
        },
      });
      return queryResponse;
    } catch (error) {
      console.error(error);
      throw new HttpException(
        `Error in Updating Sample with ID '${sampleId}' - ${error.name}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async deleteSample(sampleId: number): Promise<number> {
    try {
      const queryResponse = await this.sampleModel.destroy({
        where: {
          id: sampleId,
        },
      });
      return queryResponse;
    } catch (error) {
      console.error(error);
      throw new HttpException(
        `Error in Deleting Sample with ID '${sampleId}' - ${error.name}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}
